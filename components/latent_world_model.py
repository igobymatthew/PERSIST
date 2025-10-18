import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """Encodes an observation into a latent state."""

    def __init__(self, obs_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class TransitionModel(nn.Module):
    """Predicts the next latent state from the current latent state and action."""

    def __init__(self, latent_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent_state, action):
        x = torch.cat([latent_state, action], dim=-1)
        return self.net(x)


class Decoder(nn.Module):
    """Reconstructs an observation from a latent state."""

    def __init__(self, latent_dim, obs_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, obs_dim)
        )

    def forward(self, latent_state):
        return self.net(latent_state)


class LatentWorldModel(nn.Module):
    """A world model that learns a latent representation of the environment."""

    def __init__(self, obs_dim, act_dim, latent_dim=32, hidden_dim=128, lr=1e-3):
        super().__init__()
        self.latent_dim = latent_dim
        self.is_dreamer = False
        self.supports_imagination = False
        self.encoder = Encoder(obs_dim, latent_dim, hidden_dim)
        self.transition = TransitionModel(latent_dim, act_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, obs_dim, hidden_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_model(self, obs, act, next_obs):
        """Trains the encoder, transition model, and decoder."""
        obs = torch.as_tensor(obs, dtype=torch.float32)
        act = torch.as_tensor(act, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32)

        # 1. Encode observations
        latent_state = self.encoder(obs)
        next_latent_state_gt = self.encoder(next_obs)  # Ground truth for next latent

        # 2. Predict next latent state
        next_latent_state_pred = self.transition(latent_state, act)

        # 3. Decode from latent states to reconstruct observations
        reconstructed_obs = self.decoder(latent_state)

        # Calculate losses
        reconstruction_loss = self.loss_fn(reconstructed_obs, obs)
        dynamics_loss = self.loss_fn(
            next_latent_state_pred, next_latent_state_gt.detach()
        )

        total_loss = reconstruction_loss + dynamics_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    @torch.no_grad()
    def rollout_viability(
        self,
        initial_obs,
        initial_internal_state,
        action_sequences,
        internal_model,
        viability_approximator,
    ):
        device = next(self.parameters()).device

        obs_tensor = torch.as_tensor(initial_obs, dtype=torch.float32, device=device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        latent = self.encoder(obs_tensor)

        action_seq_tensor = torch.as_tensor(
            action_sequences, dtype=torch.float32, device=device
        )
        if action_seq_tensor.dim() == 2:
            action_seq_tensor = action_seq_tensor.unsqueeze(0)
        num_candidates, horizon, _ = action_seq_tensor.shape

        latent = latent.to(action_seq_tensor.device)
        if latent.size(0) == 1 and num_candidates > 1:
            latent = latent.repeat(num_candidates, 1)

        predicted_obs = []
        z = latent
        for t in range(horizon):
            z = self.transition(z, action_seq_tensor[:, t, :])
            predicted_obs.append(self.decoder(z))

        predicted_obs_tensor = torch.stack(predicted_obs, dim=1)

        internal_state = torch.as_tensor(
            initial_internal_state, dtype=torch.float32, device=device
        )
        if internal_state.dim() == 1:
            internal_state = internal_state.unsqueeze(0)
        if internal_state.size(0) == 1 and num_candidates > 1:
            internal_state = internal_state.repeat(num_candidates, 1)

        predicted_internal = []
        margins = []
        state = internal_state
        for t in range(horizon):
            action_t = action_seq_tensor[:, t, :]
            state = internal_model.predict_next(state, action_t)
            predicted_internal.append(state)
            margin = viability_approximator.get_margin(state)
            if margin.dim() == 1:
                margin = margin.unsqueeze(-1)
            margins.append(margin.squeeze(-1))

        predicted_internal_tensor = torch.stack(predicted_internal, dim=1)
        margins_tensor = torch.stack(margins, dim=1)

        return {
            "predicted_observations": predicted_obs_tensor,
            "predicted_internal_states": predicted_internal_tensor,
            "margins": margins_tensor,
        }

    def compute_surprise_reward(self, obs, act, next_obs):
        """Computes surprise as the reconstruction error of the next observation."""
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            act = torch.as_tensor(act, dtype=torch.float32).unsqueeze(0)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)

            # Predict the next latent state
            latent_state = self.encoder(obs)
            next_latent_pred = self.transition(latent_state, act)

            # Reconstruct the next observation from the predicted latent state
            reconstructed_next_obs = self.decoder(next_latent_pred)

            # Surprise is the error in reconstructing the actual next observation
            surprise = self.loss_fn(reconstructed_next_obs, next_obs)
        return surprise.item()
