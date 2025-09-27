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
            nn.Linear(hidden_dim, latent_dim)
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
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, latent_state, action):
        x = torch.cat([latent_state, action], dim=-1)
        return self.net(x)

class Decoder(nn.Module):
    """Reconstructs an observation from a latent state."""
    def __init__(self, latent_dim, obs_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

    def forward(self, latent_state):
        return self.net(latent_state)

class LatentWorldModel(nn.Module):
    """A world model that learns a latent representation of the environment."""
    def __init__(self, obs_dim, act_dim, latent_dim=32, hidden_dim=128, lr=1e-3):
        super().__init__()
        self.latent_dim = latent_dim
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
        next_latent_state_gt = self.encoder(next_obs) # Ground truth for next latent

        # 2. Predict next latent state
        next_latent_state_pred = self.transition(latent_state, act)

        # 3. Decode from latent states to reconstruct observations
        reconstructed_obs = self.decoder(latent_state)

        # Calculate losses
        reconstruction_loss = self.loss_fn(reconstructed_obs, obs)
        dynamics_loss = self.loss_fn(next_latent_state_pred, next_latent_state_gt.detach())

        total_loss = reconstruction_loss + dynamics_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

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